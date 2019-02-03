%
% superclass for all temporal signals
%
% author: Martin F. Schiffner
% date: 2019-02-03
% modified: 2019-02-03
%
classdef signal

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % properties
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	properties (SetAccess = private)

        % independent properties
        samples ( 1, : ) physical_values.physical_value	% temporal samples of the signal
        f_s ( 1, 1 ) physical_values.frequency          % temporal sampling rate

        % dependent properties
        T_duration
    end

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % methods
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = signal( samples, f_s )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure samples is a cell array
            if ~iscell( samples )
                samples = { samples };
            end

            % classes of samples and f_s will be checked in assignment

            % ensure equal number of dimensions and sizes
            auxiliary.mustBeEqualSize( samples, f_s );

            %--------------------------------------------------------------
            % 2.) create signals
            %--------------------------------------------------------------
            % construct column vector of objects
            N_objects = numel( samples );
            objects = repmat( objects, size( samples ) );

            % check and set independent properties
            for index_object = 1:N_objects

                % ensure row vectors
                if ~isrow( samples{ index_object } )
                    errorStruct.message     = sprintf( 'The content of samples{ %d } must be a row vector!', index_object );
                    errorStruct.identifier	= 'signal:NoRowVector';
                    error( errorStruct );
                end

                % set independent properties
                objects( index_object ).samples = samples{ index_object };
                objects( index_object ).f_s = f_s( index_object );
            end

        end % function objects = signal( samples, f_s )

    end % methods

end
