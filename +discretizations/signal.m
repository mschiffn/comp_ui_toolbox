%
% superclass for all temporal signals
%
% author: Martin F. Schiffner
% date: 2019-02-03
% modified: 2019-02-20
%
classdef signal

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % properties
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	properties (SetAccess = private)

        % independent properties
        set_t ( 1, 1 ) discretizations.set_discrete_time	% set of discrete time instants
        samples ( 1, : ) physical_values.physical_value     % temporal samples of the signal

    end

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % methods
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = signal( sets_t, samples )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure samples is a cell array
            if ~iscell( samples )
                samples = { samples };
            end

            % ensure equal number of dimensions and sizes
            auxiliary.mustBeEqualSize( sets_t, samples );

            %--------------------------------------------------------------
            % 2.) create signals
            %--------------------------------------------------------------
            % construct column vector of objects
            N_objects = numel( sets_t );
            objects = repmat( objects, size( sets_t ) );

            % check and set independent properties
            for index_object = 1:N_objects

                % ensure row vectors with suitable numbers of components
                if ~( isrow( samples{ index_object } ) && numel( samples{ index_object } ) == abs( sets_t( index_object ) ) )
                    errorStruct.message     = sprintf( 'The content of samples{ %d } must be a row vector with %d components!', index_object, abs( sets_t( index_object ) ) );
                    errorStruct.identifier	= 'signal:NoRowVector';
                    error( errorStruct );
                end

                % set independent properties
                objects( index_object ).set_t = sets_t( index_object );
                objects( index_object ).samples = samples{ index_object };

            end % for index_object = 1:N_objects

        end % function objects = signal( sets_t, samples )

    end % methods

end
