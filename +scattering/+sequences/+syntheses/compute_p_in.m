function fields = compute_p_in( sequence, varargin )
%
% compute incident acoustic pressure field
%
% author: Martin F. Schiffner
% date: 2019-03-16
% modified: 2020-01-10
%
% TODO: move to class sequence

	% print status
	time_start = tic;
	str_date_time = sprintf( '%04d-%02d-%02d: %02d:%02d:%02d', fix( clock ) );
	fprintf( '\t %s: computing incident acoustic pressure field (kappa)...', str_date_time );

	%----------------------------------------------------------------------
	% 1.) check arguments
	%----------------------------------------------------------------------
	% ensure class scattering.sequences.sequence (scalar)
	if ~( isa( sequence, 'scattering.sequences.sequence' ) && isscalar( sequence ) )
        errorStruct.message = 'sequence must be a single scattering.sequences.sequence!';
        errorStruct.identifier = 'compute_p_in:NoSpatiospectral';
        error( errorStruct );
    end

	% ensure nonempty indices_incident
	if nargin >= 2 && ~isempty( varargin{ 1 } )
        indices_incident = varargin{ 1 };
    else
        indices_incident = ( 1:numel( sequence.settings ) );
    end

	% ensure positive integers
	mustBeInteger( indices_incident );
	mustBePositive( indices_incident );

	% ensure that indices_incident do not exceed the number of sequential pulse-echo measurements
	if any( indices_incident > numel( sequence.settings ) )
        errorStruct.message = 'indices_incident must not exceed the number of sequential pulse-echo measurements!';
        errorStruct.identifier = 'compute_p_in:InvalidMeasurement';
        error( errorStruct );
    end

	%----------------------------------------------------------------------
	% 2.) compute incident acoustic pressure fields
	%----------------------------------------------------------------------
	% map unique frequencies of pulse-echo measurements to unique frequencies
	indices_f_to_unique_measurement = sequence.indices_f_to_unique( indices_incident );

	% extract transducer control settings in synthesis mode (unique frequencies)
	settings_tx_unique = reshape( [ sequence.settings( indices_incident ).tx_unique ], size( indices_incident ) );

	% extract normal velocities (unique frequencies)
	v_d_unique = reshape( [ sequence.settings( indices_incident ).v_d_unique ], size( indices_incident ) );

	% extract frequency axes (unique frequencies)
	axes_f_unique = reshape( [ v_d_unique.axis ], size( indices_incident ) );
	N_samples_f = abs( axes_f_unique );

	% specify cell array for p_incident
	p_incident = cell( size( indices_incident ) );

	% iterate selected incident waves
	for index_incident_sel = 1:numel( indices_incident )

        % index of incident wave
        index_incident = indices_incident( index_incident_sel );

        % extract
        indices_f_to_unique_act = indices_f_to_unique_measurement{ index_incident_sel };

        %------------------------------------------------------------------
        % b) superimpose quasi-(d-1)-spherical waves
        %------------------------------------------------------------------
        % initialize pressure fields with zeros
        p_incident{ index_incident_sel } = physical_values.pascal( zeros( N_samples_f( index_incident_sel ), sequence.setup.FOV.shape.grid.N_points ) );

        % iterate active array elements
        for index_active = 1:numel( settings_tx_unique( index_incident_sel ).indices_active )

            % index of active array element
            index_element = settings_tx_unique( index_incident_sel ).indices_active( index_active );

            % spatial transfer function of the active array element
            if isa( sequence.setup, 'scattering.sequences.setups.setup_grid_symmetric' )

                %----------------------------------------------------------
                % a) symmetric spatial discretization based on orthogonal regular grids
                %----------------------------------------------------------
                % shift reference spatial transfer function to infer that of the active array element
                indices_occupied_act = sequence.setup.indices_grid_FOV_shift( :, index_element );

                % extract current frequencies from unique frequencies
%                 h_tx_unique = double( sequence.h_ref.samples( indices_f_to_unique_act, indices_occupied_act ) );
                h_tx_unique = double( sequence.h_ref_aa.samples( indices_f_to_unique_act, indices_occupied_act ) );

            else

                %----------------------------------------------------------
                % b) arbitrary grid
                %----------------------------------------------------------
                % compute spatial transfer function of the active array element
                h_tx_unique = transfer_function( sequence.setup, axes_f_unique( index_incident ), index_element );

                % apply spatial anti-aliasing filter
% TODO: anti-aliasing options!
                h_tx_unique = anti_aliasing_filter( sequence.setup, h_tx_unique, operator_born.options.momentary.anti_aliasing );
                h_tx_unique = double( h_tx_unique.samples );

            end % if isa( sequence.setup, 'scattering.sequences.setups.setup_grid_symmetric' )

            % compute summand for the incident pressure field
            p_incident_summand = h_tx_unique .* double( v_d_unique( index_incident_sel ).samples( :, index_active ) );

            % add summand to the incident pressure field
% TODO: correct unit problem

            p_incident{ index_incident_sel } = p_incident{ index_incident_sel } + physical_values.pascal( p_incident_summand );

            % display result
            figure( index_incident_sel );
            test = squeeze( reshape( p_incident{ index_incident_sel }( 1, : ), sequence.setup.FOV.shape.grid.N_points_axis ) );
            if ndims( test ) == 2
                imagesc( abs( double( test ) ) );
            else
                imagesc( abs( double( squeeze( test( :, 1, : ) ) ) ) );
            end

        end % for index_active = 1:numel( settings_tx_unique( index_incident_sel ).indices_active )

    end % for index_incident_sel = 1:numel( indices_incident )

	%----------------------------------------------------------------------
	% 3.) create field objects
	%----------------------------------------------------------------------
	fields = processing.field( axes_f_unique, repmat( sequence.setup.FOV.shape.grid, size( indices_incident ) ), p_incident );

	% infer and print elapsed time
	time_elapsed = toc( time_start );
	fprintf( 'done! (%f s)\n', time_elapsed );

end % function fields = compute_p_in( sequence, varargin )
