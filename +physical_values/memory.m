%
% superclass for all memory sizes
%
% author: Martin F. Schiffner
% date: 2019-01-24
% modified: 2019-01-24
%
classdef memory < physical_values.physical_value

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	% methods
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = memory( values )

            % check number of arguments
%             if nargin ~= 1
%                 errorStruct.message     = 'The number of arguments must equal unity!';
%                 errorStruct.identifier	= 'memory:Arguments';
%                 error( errorStruct );
%             end

            % constructor of superclass
            objects@physical_values.physical_value( values );
        end % function objects = memory( values )

        %------------------------------------------------------------------
        % convert to binary unit
        %------------------------------------------------------------------
        function size_binary = convert_binary( objects, base, exponent )

            % construct column vector of sizes
            N_objects = numel( objects );
            size_binary = zeros( N_objects, 1 );

            % compute size in binary system
            for index_object = 1:N_objects
                size_binary( index_object ) = objects( index_object ).value / base^exponent;
            end

            % reshape to dimensions of the argument
            size_binary = reshape( size_binary, size( objects ) );
        end % function size_binary = convert_binary( objects, exponent )

        %------------------------------------------------------------------
        % convert to kilobyte (kB)
        %------------------------------------------------------------------
        function size_kilobyte = kilobyte( objects )

            % return result of binary conversion
            size_kilobyte = convert_binary( objects, 10, 3 );
        end % function size_kilobyte = kilobyte( objects )

        %------------------------------------------------------------------
        % convert to megabyte (MB)
        %------------------------------------------------------------------
        function size_megabyte = megabyte( objects )

            % return result of binary conversion
            size_megabyte = convert_binary( objects, 10, 6 );
        end % function size_megabyte = megabyte( objects )

        %------------------------------------------------------------------
        % convert to gigabyte (GB)
        %------------------------------------------------------------------
        function size_gigabyte = gigabyte( objects )

            % return result of binary conversion
            size_gigabyte = convert_binary( objects, 10, 9 );
        end % function size_gigabyte = gigabyte( objects )

        %------------------------------------------------------------------
        % convert to kibibyte (KiB)
        %------------------------------------------------------------------
        function size_kibibyte = kibibyte( objects )

            % return result of binary conversion
            size_kibibyte = convert_binary( objects, 2, 10 );
        end % function size_kibibyte = kibibyte( objects )

        %------------------------------------------------------------------
        % convert to mebibyte (MiB)
        %------------------------------------------------------------------
        function size_mebibyte = mebibyte( objects )

            % return result of binary conversion
            size_mebibyte = convert_binary( objects, 2, 20 );
        end % function size_mebibyte = mebibyte( objects )

        %------------------------------------------------------------------
        % convert to gibibyte (GiB)
        %------------------------------------------------------------------
        function size_gibibyte = gibibyte( objects )

            % return result of binary conversion
            size_gibibyte = convert_binary( objects, 2, 30 );
        end % function size_gibibyte = gibibyte( objects )

	end % methods

end % classdef memory
