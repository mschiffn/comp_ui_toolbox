function tof = function_tof( positions_lateral, states, varargin )
%
% compute round-trip times-of-flight for a steered plane wave
%
% positions_lateral: lateral positions of array elements
% states: state objects containing target positions and average sound speed
% tof: times-of-flight
%
% author: Martin F. Schiffner
% date: 2019-06-12
% modified: 2019-09-19

    %----------------------------------------------------------------------
    % 1.) check arguments
    %----------------------------------------------------------------------
	% ensure cell array for positions_lateral
	if ~iscell( positions_lateral )
        positions_lateral = { positions_lateral };
    end

	% ensure class calibration.state
    if ~isa( states, 'calibration.state' )
        errorStruct.message = 'states must be calibration.state!';
        errorStruct.identifier = 'function_tof:NoStates';
        error( errorStruct );
    end

    % ensure nonempty z_lens
    if nargin >= 3 && ~isempty( varargin{ 1 } )
        z_lens = varargin{ 1 };
    else
        z_lens = physical_values.meter( zeros( size( states ) ) );
    end

	% ensure nonempty c_lens
	if nargin >= 4 && ~isempty( varargin{ 2 } )
        c_lens = varargin{ 2 };
    else
        c_lens = reshape( [ states.c_avg ], size( states ) );
    end

	% multiple positions_lateral / single states
	if ~isscalar( positions_lateral ) && isscalar( states )
        states = repmat( states, size( positions_lateral ) );
    end

    % single positions_lateral / multiple states
	if isscalar( positions_lateral ) && ~isscalar( states )
        positions_lateral = repmat( positions_lateral, size( states ) );
    end

	% ensure equal number of dimensions and sizes
	auxiliary.mustBeEqualSize( positions_lateral, states, z_lens, c_lens );

	%----------------------------------------------------------------------
	% 2.) compute times-of-flight
	%----------------------------------------------------------------------
	% specify cell array for tof
	tof = cell( size( positions_lateral ) );

	% iterate states
	for index_state = 1:numel( positions_lateral )

        % compute distances
        dist_sc = vecnorm( [ positions_lateral{ index_state }, zeros( size( positions_lateral{ index_state }, 1 ), 1 ) ] - states( index_state ).position_target, 2, 2 );

        % compute propagation times
        t_in = z_lens( index_state ) / c_lens( index_state ) + ( states( index_state ).position_target( end ) - z_lens( index_state ) ) / states( index_state ).c_avg;
        t_sc_lens = z_lens( index_state ) * dist_sc / ( states( index_state ).position_target( end ) * c_lens( index_state ) );
        t_sc_water = ( 1 - z_lens( index_state ) / states( index_state ).position_target( end ) ) * dist_sc / states( index_state ).c_avg;

        % compute round-trip times-of-flight
        tof{ index_state } = t_in + t_sc_water + t_sc_lens;

    end

    % avoid cell array for single positions_lateral
	if isscalar( positions_lateral )
        tof = tof{ 1 };
    end

end % function tof = function_tof( positions_lateral, states )
