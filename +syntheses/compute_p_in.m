function fields = compute_p_in( spatiospectral, varargin )
%
% compute incident acoustic pressure field
%
% author: Martin F. Schiffner
% date: 2019-03-16
% modified: 2019-05-05
%

	%----------------------------------------------------------------------
	% 1.) check arguments
    %----------------------------------------------------------------------
    % ensure class discretizations.spatiospectral (scalar)
	if ~( isa( spatiospectral, 'discretizations.spatiospectral' ) && isscalar( spatiospectral ) )
        errorStruct.message = 'spatiospectral must be a single discretizations.spatiospectral!';
        errorStruct.identifier = 'compute_p_in:NoSpatiospectral';
        error( errorStruct );
    end

	% ensure nonempty indices_incident
	if nargin >= 3 && ~isempty( varargin{ 1 } )
        indices_incident = varargin{ 1 };
    else
        indices_incident = ( 1:numel( spatiospectral.spectral ) );
    end

	% ensure positive integers
	mustBeInteger( indices_incident );
	mustBePositive( indices_incident );

    % ensure that indices_incident do not exceed number of seq. emissions
    if any( indices_incident > numel( spatiospectral.spectral ) )
        errorStruct.message = 'spatiospectral must be a single discretizations.spatiospectral!';
        errorStruct.identifier = 'compute_p_in:NoSpatiospectral';
        error( errorStruct );
    end

	%----------------------------------------------------------------------
	% 2.) compute incident acoustic pressure fields
	%----------------------------------------------------------------------
	% extract transducer control settings in synthesis mode
	settings_tx = reshape( [ spatiospectral.spectral( indices_incident ).tx_unique ], size( spatiospectral.spectral( indices_incident ) ) );

	% extract frequency axes
	axes_f = [ settings_tx.excitation_voltages ];
    axes_f = axes_f.axis;
	N_samples_f = abs( axes_f );

	% specify cell array for p_incident
	p_incident = cell( size( indices_incident ) );

    % iterate incident waves
	for index_selected = 1:numel( indices_incident )

        % index of spectral_points
        index_incident = indices_incident( index_selected );

        %------------------------------------------------------------------
        % a) normal velocities of active elements
        %------------------------------------------------------------------
% TODO: precompute elsewhere
        v_d = settings_tx( index_incident ).excitation_voltages .* settings_tx( index_incident ).impulse_responses;

        %------------------------------------------------------------------
        % b) superimpose quasi-(d-1)-spherical waves
        %------------------------------------------------------------------
        % initialize pressure fields with zeros
        p_incident{ index_incident } = physical_values.pascal( zeros( spatiospectral.spatial.grid_FOV.N_points, N_samples_f( index_incident ) ) );

        % iterate active array elements
        for index_active = 1:numel( settings_tx( index_incident ).indices_active )

            % index of active array element
            index_element = settings_tx( index_incident ).indices_active( index_active );

            % spatial transfer function of the active array element
            if isa( spatiospectral.spatial, 'discretizations.spatial_grid_symmetric' )

                %----------------------------------------------------------
                % a) symmetric spatial discretization based on orthogonal regular grids
                %----------------------------------------------------------
%                 indices_occupied_act = spatiospectral.indices_grid_FOV_shift( :, index_element );
%                 h_tx = spatiospectral.h_ref.samples( indices_occupied_act, : );
                % shift reference spatial transfer function to infer that of the active array element
                h_tx = shift_lateral( spatiospectral.h_ref, spatiospectral.spatial, index_element );

            else

                %----------------------------------------------------------
                % b) arbitrary grid
                %----------------------------------------------------------
                % compute spatial transfer function of the active array element
                h_tx = discretizations.spatial_transfer_function( spatiospectral.spatial, spatiospectral.spectral( index_incident ), index_element );

            end % if isa( spatiospectral.spatial, 'discretizations.spatial_grid_symmetric' )

            % compute summand for the incident pressure field
            p_incident_summand = h_tx.samples .* v_d.samples( index_active, : );

            % add summand to the incident pressure field
% TODO: correct unit problem
            p_incident{ index_incident } = p_incident{ index_incident } + physical_values.pascal( double( p_incident_summand ) );
            figure(1);imagesc( squeeze( abs( double( reshape( p_incident{ index_incident }( :, 1 ), [512,512] ) ) ) ) );

        end % for index_active = 1:numel( settings_tx( index_incident ).indices_active )

    end % for index_selected = 1:numel( indices_incident )

	%----------------------------------------------------------------------
	% 3.) create field objects
	%----------------------------------------------------------------------
	fields = discretizations.field( axes_f, repmat( spatiospectral.spatial.grid_FOV, size( axes_f ) ), p_incident );

end % function fields = compute_p_in( spatiospectral, varargin )
