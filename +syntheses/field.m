%
% superclass for all fields
%
% author: Martin F. Schiffner
% date: 2019-01-22
% modified: 2019-03-20
%
classdef field

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % properties
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	properties

        % independent properties
        set_f ( 1, 1 ) discretizations.set_discrete_frequency
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
        function objects = field( sets_discrete_frequencies, samples )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            %
            if nargin == 0
                return;
            end

            % ensure class discretizations.set_discrete_frequency
            if ~isa( sets_discrete_frequencies, 'discretizations.set_discrete_frequency' )
                errorStruct.message     = 'sets_discrete_frequencies must be discretizations.set_discrete_frequency!';
                errorStruct.identifier	= 'field:NoSetDiscreteFrequency';
                error( errorStruct );
            end

            % ensure cell array for samples
            if nargin >= 2 && ~iscell( samples )
                samples = { samples };
            else
                
            end

            % ensure equal number of dimensions and sizes
            auxiliary.mustBeEqualSize( sets_discrete_frequencies, samples );

            %--------------------------------------------------------------
            % 2.) create fields
            %--------------------------------------------------------------
            objects = repmat( objects, size( sets_discrete_frequencies ) );

            %--------------------------------------------------------------
            % 3.) set independent and dependent properties
            %--------------------------------------------------------------
            for index_object = 1:numel( sets_discrete_frequencies )

                % number of discrete frequencies
                N_dimensions_act = ndims( samples{ index_object } );
                size_act = size( samples{ index_object } );
                N_samples_f = size_act( end );
                N_points = prod( size_act( 1:(end - 1) ) );
                firstdims = repmat( {':'}, 1, N_dimensions_act - 1 );

                % initialize field values with zeros
                objects( index_object ).values = cell( 1, N_samples_f );
                for index_f = 1:N_samples_f
                    objects( index_object ).values{ index_f } = samples{ index_object }( firstdims{ : }, index_f );
                end

                % compute memory consumption
                objects( index_object ).size_bytes = physical_values.memory( N_samples_f * N_points * 16 );

            end % for index_object = 1:numel( sets_discrete_frequencies )

        end % function objects = field( sets_discrete_frequencies, samples )

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
