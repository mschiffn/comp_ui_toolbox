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
	if nargin >= 2 && ~isempty( varargin{ 1 } )
        indices_incident = varargin{ 1 };
    else
        indices_incident = ( 1:numel( spatiospectral.spectral ) );
    end

	% ensure positive integers
	mustBeInteger( indices_incident );
	mustBePositive( indices_incident );

	% ensure that indices_incident do not exceed the number of sequential pulse-echo measurements
	if any( indices_incident > numel( spatiospectral.spectral ) )
        errorStruct.message = 'indices_incident must not exceed the number of sequential pulse-echo measurements!';
        errorStruct.identifier = 'compute_p_in:InvalidMeasurement';
        error( errorStruct );
    end

	%----------------------------------------------------------------------
	% 2.) compute incident acoustic pressure fields
	%----------------------------------------------------------------------
	% extract transducer control settings in synthesis mode (unique frequencies)
	settings_tx_unique = reshape( [ spatiospectral.spectral( indices_incident ).tx_unique ], size( indices_incident ) );

	% extract normal velocities (unique frequencies)
	v_d_unique = reshape( [ spatiospectral.spectral( indices_incident ).v_d_unique ], size( indices_incident ) );

	% extract frequency axes (unique frequencies)
	axes_f_unique = reshape( [ v_d_unique.axis ], size( indices_incident ) );
	N_samples_f = abs( axes_f_unique );

	% specify cell array for p_incident
	p_incident = cell( size( indices_incident ) );

	% iterate selected incident waves
	for index_incident_sel = 1:numel( indices_incident )

        % index of incident wave
        index_incident = indices_incident( index_incident_sel );

        %------------------------------------------------------------------
        % b) superimpose quasi-(d-1)-spherical waves
        %------------------------------------------------------------------
        % initialize pressure fields with zeros
        p_incident{ index_incident_sel } = physical_values.pascal( zeros( spatiospectral.spatial.grid_FOV.N_points, N_samples_f( index_incident_sel ) ) );

        % iterate active array elements
        for index_active = 1:numel( settings_tx_unique( index_incident_sel ).indices_active )

            % index of active array element
            index_element = settings_tx_unique( index_incident_sel ).indices_active( index_active );

            % spatial transfer function of the active array element
            if isa( spatiospectral.spatial, 'discretizations.spatial_grid_symmetric' )

                %----------------------------------------------------------
                % a) symmetric spatial discretization based on orthogonal regular grids
                %----------------------------------------------------------
                % shift reference spatial transfer function to infer that of the active array element
%                 indices_occupied_act = spatiospectral.indices_grid_FOV_shift( :, indices_active_rx_to_unique{ index_mix }( index_active ) );

                % extract current frequencies from unique frequencies
%                 h_tx_unique = spatiospectral.h_ref( index_incident ).samples( indices_occupied_act, : );
                h_tx_unique = shift_lateral( spatiospectral.h_ref( index_incident ), spatiospectral.spatial, index_element );

            else

                %----------------------------------------------------------
                % b) arbitrary grid
                %----------------------------------------------------------
                % compute spatial transfer function of the active array element
                h_tx_unique = discretizations.spatial_transfer_function( spatiospectral.spatial, spatiospectral.spectral( index_incident ), index_element );

            end % if isa( spatiospectral.spatial, 'discretizations.spatial_grid_symmetric' )

            % compute summand for the incident pressure field
            p_incident_summand = double( h_tx_unique.samples ) .* double( v_d_unique( index_incident_sel ).samples( index_active, : ) );

            % add summand to the incident pressure field
% TODO: correct unit problem
            p_incident{ index_incident_sel } = p_incident{ index_incident_sel } + physical_values.pascal( p_incident_summand );
            figure(index_incident_sel);imagesc( abs( double( reshape( p_incident{ index_incident_sel }( :, 1 ), [512,512] ) ) ) );

        end % for index_active = 1:numel( settings_tx_unique( index_incident_sel ).indices_active )

    end % for index_incident_sel = 1:numel( indices_incident )

	%----------------------------------------------------------------------
	% 3.) create field objects
	%----------------------------------------------------------------------
	fields = discretizations.field( axes_f_unique, repmat( spatiospectral.spatial.grid_FOV, size( indices_incident ) ), p_incident );

end % function fields = compute_p_in( spatiospectral, varargin )
