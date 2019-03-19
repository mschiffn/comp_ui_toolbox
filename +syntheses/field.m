%
% superclass for all fields
%
% author: Martin F. Schiffner
% date: 2019-01-22
% modified: 2019-03-19
%
classdef field

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % properties
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	properties

        % independent properties
        values % phasors of the acoustic value for each grid point in the FOV

        % dependent properties
        size_bytes ( 1, 1 ) physical_values.memory	% memory consumption (B)

    end % properties

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % methods
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = field( spatiospectral )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % return if no input argument
            if nargin == 0
                return;
            end

            % ensure class discretizations.spatiospectral
            if ~( isa( spatiospectral, 'discretizations.spatiospectral' ) && numel( spatiospectral ) == 1 )
                errorStruct.message     = 'spatiospectral must be a single discretizations.spatiospectral!';
                errorStruct.identifier	= 'field:NoDiscretization';
                error( errorStruct );
            end

            % ensure class discretizations.spatial_grid
            if ~isa( spatiospectral.spatial, 'discretizations.spatial_grid' )
                errorStruct.message     = 'spatiospectral.spatial must be discretizations.spatial_grid!';
                errorStruct.identifier	= 'field:NoSpatialGrid';
                error( errorStruct );
            end

            % ensure class discretizations.spectral_points
            if ~isa( spatiospectral.spectral, 'discretizations.spectral_points' )
                errorStruct.message     = 'spatiospectral.spectral must be discretizations.spectral_points!';
                errorStruct.identifier	= 'field:NoSpectralPoints';
                error( errorStruct );
            end

            %--------------------------------------------------------------
            % 2.) construct objects
            %--------------------------------------------------------------
            % construct column vector of objects
            objects = repmat( objects, size( spatiospectral.spectral ) );

            %--------------------------------------------------------------
            % 3.) initialize objects with zeros / compute memory consumption
            %--------------------------------------------------------------
            for index_object = 1:numel( spatiospectral.spectral )

                % number of unique discrete frequencies
                N_samples_f = abs( spatiospectral.spectral( index_object ).tx_unique.excitation_voltages( 1 ).set_f );
                N_points = spatiospectral.spatial.grid_FOV.N_points;

                % initialize field values with zeros
                objects( index_object ).values = cell( 1, N_samples_f );
                for index_f = 1:N_samples_f
                    objects( index_object ).values{ index_f } = zeros( spatiospectral.spatial.grid_FOV.N_points_axis );                   
                end

                % compute memory consumption
                objects( index_object ).size_bytes = physical_values.memory( N_samples_f * N_points * 16 );

            end % for index_object = 1:numel( spatiospectral.spectral )

        end % function objects = field( spatiospectral )

        %------------------------------------------------------------------
        % show
        %------------------------------------------------------------------
        function hdl = show( fields )

            %--------------------------------------------------------------
            % 1.) display fields
            %--------------------------------------------------------------
            hdl = zeros( size( fields ) );
            for index_object = 1:numel( fields )

                N_samples_f = numel( fields( index_object ).values );
                index_ctr = round( N_samples_f / 2 );

                hdl( index_object ) = figure( index_object );

                % check number of spatial dimensions
                switch ndims( fields( index_object ).values{ 1 } )

                    case 2

                        %--------------------------------------------------
                        % a) two-dimensional Euclidean space
                        %--------------------------------------------------
                        subplot( 2, 3, 1);
                        imagesc( abs( fields( index_object ).values{ 1 } ) );
                        subplot( 2, 3, 2);
                        imagesc( abs( fields( index_object ).values{ index_ctr } ) );
                        subplot( 2, 3, 3);
                        imagesc( abs( fields( index_object ).values{ end } ) );
                        subplot( 2, 3, 4);
                        imagesc( angle( fields( index_object ).values{ 1 } ) );
                        subplot( 2, 3, 5);
                        imagesc( angle( fields( index_object ).values{ index_ctr } ) );
                        subplot( 2, 3, 6);
                        imagesc( angle( fields( index_object ).values{ end } ) );

                    case 3

                        %--------------------------------------------------
                        % b) three-dimensional Euclidean space
                        %--------------------------------------------------
                        subplot( 3, 3, 1);
                        imagesc( abs( squeeze( fields( index_object ).values{ 1 }( :, 1, :) ) ) );
                        subplot( 3, 3, 2);
                        imagesc( abs( squeeze( fields( index_object ).values{ index_ctr }( :, 1, :) ) ) );
                        subplot( 3, 3, 3);
                        imagesc( abs( squeeze( fields( index_object ).values{ end }( :, 1, :) ) ) );
                        subplot( 3, 3, 4);
                        imagesc( abs( squeeze( fields( index_object ).values{ 1 }( :, :, 1) ) ) );
                        subplot( 3, 3, 5);
                        imagesc( abs( squeeze( fields( index_object ).values{ index_ctr }( :, :, 1) ) ) );
                        subplot( 3, 3, 6);
                        imagesc( abs( squeeze( fields( index_object ).values{ end }( :, :, 1) ) ) );
                        subplot( 3, 3, 7);
                        imagesc( abs( squeeze( fields( index_object ).values{ 1 }( 1, :, :) ) ) );
                        subplot( 3, 3, 8);
                        imagesc( abs( squeeze( fields( index_object ).values{ index_ctr }( 1, :, :) ) ) );
                        subplot( 3, 3, 9);
                        imagesc( abs( squeeze( fields( index_object ).values{ end }( 1, :, :) ) ) );

                    otherwise

                        %--------------------------------------------------
                        % c) dimensionality not implemented
                        %--------------------------------------------------
                        errorStruct.message     = 'Number of dimensions not implemented!';
                        errorStruct.identifier	= 'show:UnknownDimensions';
                        error( errorStruct );

                end % switch ndims( fields( index_object ).values{ 1 } )
                
            end % for index_object = 1:numel( fields )

        end % function hdl = show( fields )

	end % methods

end % classdef field
