%
% superclass for all Fourier transforms
%
% author: Martin F. Schiffner
% date: 2019-02-23
% modified: 2019-02-23
%
classdef fourier_transform

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % properties
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	properties (SetAccess = private)

        % independent properties
        set_f ( 1, 1 ) discretizations.set_discrete_frequency	% set of discrete time instants
        samples %( 1, : ) physical_values.physical_value     % Fourier series coefficients

    end

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % methods
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = fourier_transform( sets_f, samples )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure samples is a cell array
            if ~iscell( samples )
                samples = { samples };
            end

            % ensure equal number of dimensions and sizes
            auxiliary.mustBeEqualSize( sets_f, samples );

            %--------------------------------------------------------------
            % 2.) create Fourier transforms
            %--------------------------------------------------------------
            % construct objects
            objects = repmat( objects, size( sets_f ) );

            % check and set independent properties
            for index_object = 1:numel( sets_f )

                % ensure row vectors with suitable numbers of components
                if ~( isrow( samples{ index_object } ) && numel( samples{ index_object } ) == abs( sets_f( index_object ) ) )
                    errorStruct.message     = sprintf( 'The content of samples{ %d } must be a row vector with %d components!', index_object, abs( sets_f( index_object ) ) );
                    errorStruct.identifier	= 'fourier_transform:NoRowVector';
                    error( errorStruct );
                end

                % set independent properties
                objects( index_object ).set_f = sets_f( index_object );
                objects( index_object ).samples = samples{ index_object };

            end % for index_object = 1:numel( sets_f )

        end % function objects = fourier_transform( sets_f, samples )

        %------------------------------------------------------------------
        % 2-D line plot (overload plot function)
        %------------------------------------------------------------------
        function objects = plot( objects )

            % create new figure
            figure;

            % plot all signals in single figure
            plot( double( objects( 1 ).set_f.S ), abs( objects( 1 ).samples ) );
            hold on;
            for index_object = 2:numel( objects )
                plot( double( objects( index_object ).set_f.S ), abs( objects( index_object ).samples ) );
            end % for index_object = 2:numel( objects )
            hold off;

        end % function objects = plot( objects )

    end % methods

end
